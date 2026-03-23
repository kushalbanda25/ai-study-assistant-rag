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
#  GLOBAL STYLES (Premium ChatGPT-like)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e6edf3;
    font-family: 'Inter', -apple-system, sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #8b949e !important; }

/* ── Title ── */
h1 { color: #ffffff !important; font-weight: 700; letter-spacing: -0.05rem; }

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 12px 16px;
    margin-bottom: 12px;
    border: 1px solid transparent;
}

/* User bubble */
div[data-testid="stChatMessage"]:has(img[alt="user"]) {
    background: #1e293b;
    border: 1px solid #334155;
    margin-left: 20%;
}

/* Assistant bubble */
div[data-testid="stChatMessage"]:has(img[alt="assistant"]) {
    background: #0d1117;
    border: 1px solid #30363d;
    margin-right: 15%;
}

/* ── Chat Input ── */
[data-testid="stChatInputTextArea"] {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    color: #e6edf3 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 8px;
    font-weight: 500;
}
.stButton > button:hover {
    background: #30363d;
    border-color: #8b949e;
    color: #ffffff;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL CACHE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing AI components...")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
STORAGE_DIR        = "storage"
FAISS_PATH         = os.path.join(STORAGE_DIR, "index.faiss")
CHUNKS_PATH        = os.path.join(STORAGE_DIR, "chunks.pkl")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# Fallback sequence for robustness (skips on 400, 404, 422, 429)
LLM_MODELS = [
    "deepseek/deepseek-chat:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
]

MAX_CONTEXT_WORDS  = 600
TOP_K_CHUNKS       = 3
MAX_HISTORY_TURNS  = 8

# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────
def sanitize_text(text: str) -> str:
    """Strips control characters that break JSON/API requests."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


def read_pdf(file) -> str:
    text = ""
    try:
        pdf = PdfReader(file)
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + " "
    except Exception as e:
        st.sidebar.error(f"Read error: {e}")
    return sanitize_text(re.sub(r'\s+', ' ', text).strip())


def chunk_text(text: str, chunk_size: int = 150) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# ─────────────────────────────────────────────
#  RAG & LLM
# ─────────────────────────────────────────────
def get_context(query: str) -> str:
    if not st.session_state.index: return ""
    
    query_vec = model.encode([query])
    k = min(TOP_K_CHUNKS, len(st.session_state.chunks))
    _, I = st.session_state.index.search(np.array(query_vec), k=k)
    
    relevant = [st.session_state.chunks[i] for i in I[0] if i < len(st.session_state.chunks)]
    return " ".join(relevant)[:MAX_CONTEXT_WORDS * 5]


def build_messages(context: str, question: str, history: list[dict]) -> list[dict]:
    # Clean, strict instructions for high-quality responses
    system_instr = (
        "You are a Senior AI Study Assistant. Help users learn from their material.\n\n"
        "RULES:\n"
        "- If context is provided, answer ONLY using that context.\n"
        "- If the answer is missing from context, say: \"I couldn't find that in your notes.\"\n"
        "- Structure answers with Markdown: bold headers, bullet points, or numbering.\n"
        "- Be professional, clear, and direct. Avoid unnecessary conversational filler.\n"
        "- If referring to specific parts of the context, be precise.\n\n"
        f"STUDY MATERIAL CONTEXT:\n{context}"
    )
    
    # Universal compatibility: Inject system prompt correctly
    msgs = [{"role": "system", "content": system_instr}]
    # Add trimmed conversation history (Last N messages)
    msgs.extend(history[-(MAX_HISTORY_TURNS * 2):])
    # Add current question
    msgs.append({"role": "user", "content": question})
    return msgs


def _call_api(model_id: str, messages: list, api_key: str) -> requests.Response:
    return requests.post(
        url=OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "http://localhost:8501",
            "X-Title":       "AI Study Assistant",
        },
        json={
            "model": model_id,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        timeout=15
    )


def ask_llm(context: str, query: str, history: list[dict]) -> str:
    # Get key from secrets (Cloud) or env (Local)
    api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", "")).strip()
    if not api_key:
        return "⚠️ **API Key missing.** Please set `OPENROUTER_API_KEY` in Streamlit secrets or environment."

    messages = build_messages(context, query, history)
    
    last_err_info = "Status Unknown"
    
    # Fallback Loop: Skips transient errors (400, 404, 422, 429)
    for model_id in LLM_MODELS:
        try:
            resp = _call_api(model_id, messages, api_key)
            
            # If 401 (Unauthorized) → fail immediately (bad key)
            if resp.status_code == 401:
                return "🔑 **Invalid API key.** Please check your `OPENROUTER_API_KEY`."
                
            # If model/data error (400, 404, 422, 429) → try next
            if resp.status_code in (400, 404, 422, 429):
                try:
                    data = resp.json().get("error", {}).get("message", f"Status {resp.status_code}")
                    last_err_info = f"Model `{model_id}` → {data}"
                except:
                    last_err_info = f"Model `{model_id}` → Status {resp.status_code}"
                continue
                
            resp.raise_for_status()
            data = resp.json()
            
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"].strip()
                
            last_err_info = f"Model `{model_id}` → Empty response"
            continue 
            
        except Exception as e:
            last_err_info = f"Model `{model_id}` → Connection Error: {str(e)[:50]}"
            continue 
            
    return (
        f"🚀 **All models are currently busy.**\n\n"
        f"**Technical Details:** {last_err_info}\n\n"
        "Please wait 10s and try again, or check your API key and balance."
    )


# ─────────────────────────────────────────────
#  SESSION & PERSISTENCE
# ─────────────────────────────────────────────
if "index"    not in st.session_state: st.session_state.index = None
if "chunks"   not in st.session_state: st.session_state.chunks = []
if "messages" not in st.session_state: st.session_state.messages = []
if "ready"    not in st.session_state: st.session_state.ready = False

# Auto-restore on start
if not st.session_state.ready and os.path.exists(FAISS_PATH):
    try:
        st.session_state.index = faiss.read_index(FAISS_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            st.session_state.chunks = pickle.load(f)
        st.session_state.ready = True
    except: pass

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Your Study Library")
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        do_process = st.button("🚀 Process", use_container_width=True)
    with col_b:
        do_clear = st.button("🗑️ Clear", use_container_width=True)

    if do_clear:
        for p in [FAISS_PATH, CHUNKS_PATH]:
            if os.path.exists(p): os.remove(p)
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.messages = []
        st.session_state.ready = False
        st.rerun()

    if do_process and uploaded:
        all_text = ""
        with st.spinner("Extracting knowledge..."):
            for f in uploaded:
                all_text += read_pdf(f) + " "
        
        if all_text.strip():
            chunks = chunk_text(all_text)
            with st.spinner(f"Indexing {len(chunks)} fragments..."):
                vecs = model.encode(chunks)
                idx = faiss.IndexFlatL2(vecs.shape[1])
                idx.add(np.array(vecs))
            
            st.session_state.index = idx
            st.session_state.chunks = chunks
            st.session_state.ready = True
            
            os.makedirs(STORAGE_DIR, exist_ok=True)
            faiss.write_index(idx, FAISS_PATH)
            with open(CHUNKS_PATH, "wb") as f:
                pickle.dump(chunks, f)
            st.success(f"Indexed {len(uploaded)} files!")
        else:
            st.error("No readable text found in PDFs.")

    st.markdown("---")
    if st.session_state.ready:
        st.markdown(f"✅ **{len(st.session_state.chunks)}** notes fragments loaded.")
    else:
        st.markdown("⚠️ No notes indexed.")
    
    if st.button("🧹 Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
#  MAIN LOGIC
# ─────────────────────────────────────────────
st.title("📘 AI Study Assistant")
st.markdown("<p style='color:#6e7681; font-size: 1.1rem; margin-top:-1rem;'>Upload notes to chat with your materials</p>", unsafe_allow_html=True)

# Initial Help
if not st.session_state.ready:
    st.info("👋 **Welcome!** Start by uploading your PDF study notes in the sidebar and clicking **Process**.")

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Local Greeting detection
GREETINGS = ["hi", "hello", "hey", "who are you", "what are you", "thanks", "thank you"]

# Chat Input
query = st.chat_input("Ask a question about your notes...")

if query:
    # ── Immediate render for user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── Logic
    q_low = query.lower().strip().rstrip("!?.")
    
    if q_low in GREETINGS:
        ans = "👋 Hello! I'm your AI Study Assistant. Upload some PDFs and I'll help you master the material!"
        if q_low in ["thanks", "thank you"]: ans = "You're very welcome! Happy studying! 📚"
    elif not st.session_state.ready:
        ans = "⚠️ I need your study notes first! Please **upload and process** PDFs in the sidebar."
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing notes..."):
                ctx = get_context(query)
                hist = st.session_state.messages[:-1]
                ans = ask_llm(ctx, query, hist)
    
    # Render and store assistant reply
    if q_low in GREETINGS or not st.session_state.ready:
        with st.chat_message("assistant"):
            st.markdown(ans)
    else:
        st.markdown(ans)
        
    st.session_state.messages.append({"role": "assistant", "content": ans})