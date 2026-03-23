import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import re
import pickle
import os
import time
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
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #0f0f0f;
    color: #ececec;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: #1a1a1a;
    border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] * { color: #d1d1d1 !important; }
h1 { color: #ffffff !important; font-weight: 700; letter-spacing: -0.5px; }
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 6px;
}
div[data-testid="stChatMessage"]:has(img[alt="user"])      { background: #1e1e2e; }
div[data-testid="stChatMessage"]:has(img[alt="assistant"]) { background: #161b22; }
[data-testid="stChatInputTextArea"] {
    background: #1e1e1e !important;
    color: #ececec !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 12px !important;
}
.stButton > button {
    background: #2d2d2d; color: #ececec;
    border: 1px solid #3a3a3a; border-radius: 8px;
    font-size: 0.85rem; transition: background 0.2s;
}
.stButton > button:hover { background: #3d3d3d; }
.stAlert { border-radius: 10px; font-size: 0.9rem; }
.stSpinner > div { color: #a0a0a0 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PROVIDER CONFIG  (verified March 2026)
# ─────────────────────────────────────────────

# ── OpenRouter ──────────────────────────────
#  URL  : https://openrouter.ai/api/v1/chat/completions
#  Auth : Bearer <OPENROUTER_API_KEY>
#  Free models verified at openrouter.ai/collections/free-models
OPENROUTER_URL    = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = [
    "openrouter/free",                          # auto-picks best free model
    "meta-llama/llama-3.3-70b-instruct:free",   # reliable, widely used
    "nvidia/nemotron-3-super-120b-a12b:free",   # 262K ctx, strong reasoning
    "arcee-ai/trinity-large-preview:free",      # 400B, 128K ctx
]

# ── Groq ────────────────────────────────────
#  URL  : https://api.groq.com/openai/v1/chat/completions
#  Auth : Bearer <GROQ_API_KEY>
#  Free : no credit card needed — very fast (LPU hardware)
#         Llama 3.3 70B: ~6K tokens/min, 500K tokens/day
#         Llama 3.1 8B : ~30K tokens/min, 1M tokens/day
#  Get key at: console.groq.com
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # best quality on Groq free tier
    "llama-3.1-8b-instant",      # fastest, higher rate limit
    "qwen/qwen3-32b",            # Qwen3 32B (March 2026)
]

# ── Shared constants ────────────────────────
STORAGE_DIR       = "storage"
FAISS_PATH        = os.path.join(STORAGE_DIR, "index.faiss")
CHUNKS_PATH       = os.path.join(STORAGE_DIR, "chunks.pkl")
MAX_CONTEXT_WORDS = 600
TOP_K_CHUNKS      = 3
MAX_HISTORY_TURNS = 6
RETRY_WAIT_SECS   = 6

PROVIDER_OPTIONS  = ["OpenRouter (free)", "Groq (free)"]

# ─────────────────────────────────────────────
#  MODEL CACHE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

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
    index  = st.session_state.index
    chunks = st.session_state.chunks
    query_vec = embed_model.encode([query])
    k = min(TOP_K_CHUNKS, len(chunks))
    _, I = index.search(np.array(query_vec), k=k)
    selected = [chunks[i] for i in I[0] if i < len(chunks)]
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
#  LLM HELPERS
# ─────────────────────────────────────────────
def build_messages(context: str, question: str, history: list[dict]) -> list[dict]:
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
    messages.extend(history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": question})
    return messages


def _get_secret(key: str) -> str:
    """Read from Streamlit Secrets first, then env variable."""
    val = ""
    try:
        val = st.secrets.get(key, "")
    except Exception:
        pass
    if not val:
        val = os.getenv(key, "")
    return val.strip()


def _openai_compat_post(url: str, model_id: str, messages: list,
                        api_key: str, extra_headers: dict | None = None) -> requests.Response:
    """Generic OpenAI-compatible POST (used by OpenRouter, Groq, and HF chat endpoint)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    return requests.post(
        url=url,
        headers=headers,
        json={
            "model":       model_id,
            "messages":    messages,
            "temperature": 0.3,
            "max_tokens":  1024,
        },
        timeout=45,
    )


def _parse_openai_response(response: requests.Response) -> str | None:
    """Return content string if valid, else None."""
    result = response.json()
    if "choices" in result and result["choices"]:
        return result["choices"][0]["message"]["content"].strip()
    return None


# ── Provider: OpenRouter ──────────────────────
def ask_openrouter(context: str, question: str, history: list[dict]) -> str:
    api_key = _get_secret("OPENROUTER_API_KEY")
    if not api_key:
        return (
            "⚠️ **OpenRouter API key not set.**\n\n"
            "In Streamlit Secrets add:\n```toml\nOPENROUTER_API_KEY = \"sk-or-v1-...\"\n```\n"
            "Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys)"
        )
    messages = build_messages(context, question, history)
    for model_id in OPENROUTER_MODELS:
        try:
            resp = _openai_compat_post(
                url=OPENROUTER_URL,
                model_id=model_id,
                messages=messages,
                api_key=api_key,
                extra_headers={
                    "HTTP-Referer": "https://ai-study-assistant.streamlit.app",
                    "X-Title":      "AI Study Assistant",
                },
            )
            if resp.status_code == 404: continue
            if resp.status_code == 429:
                time.sleep(RETRY_WAIT_SECS); continue
            resp.raise_for_status()
            content = _parse_openai_response(resp)
            if content:
                return content
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.ConnectionError:
            return "🌐 **Network error.** Check your connection."
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 401: return "🔑 **Invalid OpenRouter API key.**"
            if status in (404, 429):
                if status == 429: time.sleep(RETRY_WAIT_SECS)
                continue
            return f"⚠️ **HTTP {status}:** {e}"
        except Exception as e:
            return f"⚠️ **Unexpected error:** {e}"
    return "⚠️ **All OpenRouter free models are rate-limited.** Wait a minute and retry."


# ── Provider: Groq ────────────────────────────
def ask_groq(context: str, question: str, history: list[dict]) -> str:
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        return (
            "⚠️ **Groq API key not set.**\n\n"
            "In Streamlit Secrets add:\n```toml\nGROQ_API_KEY = \"gsk_...\"\n```\n"
            "Get a **free** key (no credit card) at [console.groq.com](https://console.groq.com)"
        )
    messages = build_messages(context, question, history)
    for model_id in GROQ_MODELS:
        try:
            resp = _openai_compat_post(
                url=GROQ_URL,
                model_id=model_id,
                messages=messages,
                api_key=api_key,
            )
            if resp.status_code == 404: continue
            if resp.status_code == 429:
                time.sleep(RETRY_WAIT_SECS); continue
            resp.raise_for_status()
            content = _parse_openai_response(resp)
            if content:
                return content
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.ConnectionError:
            return "🌐 **Network error.** Check your connection."
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 401: return "🔑 **Invalid Groq API key.**"
            if status in (404, 429):
                if status == 429: time.sleep(RETRY_WAIT_SECS)
                continue
            return f"⚠️ **HTTP {status}:** {e}"
        except Exception as e:
            return f"⚠️ **Unexpected error:** {e}"
    return "⚠️ **All Groq models are rate-limited.** Wait a minute and retry."




# ── Router ────────────────────────────────────
def ask_llm(context: str, question: str, history: list[dict]) -> str:
    provider = st.session_state.get("provider", PROVIDER_OPTIONS[0])
    if provider.startswith("OpenRouter"):
        return ask_openrouter(context, question, history)
    elif provider.startswith("Groq"):
        return ask_groq(context, question, history)
    else:
        return "⚠️ Unknown provider. Please check your selection."


# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in [
    ("index",    None),
    ("chunks",   []),
    ("messages", []),
    ("loaded",   False),
    ("provider", PROVIDER_OPTIONS[0]),
]:
    if key not in st.session_state:
        st.session_state[key] = default

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
    st.markdown("## 📘 AI Study Assistant")
    st.markdown("---")

    # ── Provider selector ──
    st.markdown("### 🤖 AI Provider")
    selected_provider = st.selectbox(
        label="Choose provider",
        options=PROVIDER_OPTIONS,
        index=PROVIDER_OPTIONS.index(st.session_state.provider),
        label_visibility="collapsed",
    )
    st.session_state.provider = selected_provider

    # ── Provider info card ──
    if selected_provider.startswith("OpenRouter"):
        st.info(
            "**OpenRouter** — free auto-router\n\n"
            "Key: `OPENROUTER_API_KEY`\n"
            "Get key → [openrouter.ai/keys](https://openrouter.ai/keys)\n\n"
            "Models: openrouter/free → Llama 3.3 70B → Nemotron Super → Trinity Large"
        )
    elif selected_provider.startswith("Groq"):
        st.info(
            "**Groq** — fastest free inference (LPU)\n\n"
            "Key: `GROQ_API_KEY`\n"
            "Get key → [console.groq.com](https://console.groq.com) *(no credit card)*\n\n"
            "Models: Llama 3.3 70B → Llama 3.1 8B → Qwen3 32B\n"
            "Limits: 500K tokens/day free"
        )

    st.markdown("---")

    # ── PDF upload ──
    st.markdown("### 📂 Upload Notes")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("⚙️ Process", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        for path in [FAISS_PATH, CHUNKS_PATH]:
            if os.path.exists(path):
                os.remove(path)
        st.session_state.index    = None
        st.session_state.chunks   = []
        st.session_state.messages = []
        st.session_state.loaded   = False
        st.success("✅ Data cleared.")

    if process_btn and uploaded_files:
        all_text = ""
        with st.spinner("Reading PDFs…"):
            for file in uploaded_files:
                all_text += read_pdf(file) + " "
        if all_text.strip():
            chunks = chunk_text(all_text)
            with st.spinner(f"Indexing {len(chunks)} chunks…"):
                embeddings = embed_model.encode(chunks, show_progress_bar=False)
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
            st.success(f"✅ {len(uploaded_files)} file(s) — {len(chunks)} chunks ready.")
        else:
            st.warning("No readable text found in the uploaded PDFs.")
    elif process_btn:
        st.info("Please upload at least one PDF first.")

    st.markdown("---")
    if st.session_state.index is not None:
        st.markdown(
            f"<small>📚 {len(st.session_state.chunks)} chunks loaded</small>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<small>No document loaded yet.</small>", unsafe_allow_html=True)

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

if st.session_state.index is None:
    st.info("⬅️ Upload your PDF notes and click **⚙️ Process** to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
#  CHAT INPUT
# ─────────────────────────────────────────────
query = st.chat_input("Ask something from your notes…")

GREETING_TRIGGERS = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "ty", "thx",
    "bye", "goodbye", "see you",
    "who are you", "what are you", "what can you do",
}

def is_greeting(text: str) -> str | None:
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
    canned = is_greeting(query)
    if canned:
        st.session_state.messages.append({"role": "user",      "content": query})
        st.session_state.messages.append({"role": "assistant", "content": canned})
        with st.chat_message("user"):      st.markdown(query)
        with st.chat_message("assistant"): st.markdown(canned)
    elif st.session_state.index is None:
        st.warning("⬅️ Please upload and process your PDFs first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                context = retrieve_context(query)
                history = st.session_state.messages[:-1]
                answer  = ask_llm(context, query, history)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})