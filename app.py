import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import re
import pickle
import os
import requests

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Study Assistant", layout="wide")

# -------------------- STYLING --------------------
st.markdown("""
<style>
.main {
    max-width: 800px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------- FUNCTIONS --------------------
def read_pdf(file):
    text = ""
    pdf = PdfReader(file)

    for page in pdf.pages:
        content = page.extract_text()
        if content:
            text += content + " "

    # Clean text
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def chunk_text(text, chunk_size=80):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def ask_llm(context, question):
    prompt = f"""
You are a smart assistant.

Rules:
- If the question is a casual greeting (like "hi" or "hello"), just greet the user back politely and DO NOT use or mention the context at all.
- Otherwise, answer ONLY based on the context provided.
- Be clear and concise
- Use bullet points
- Do NOT repeat information
- Do NOT add unnecessary explanation
- If asking for definition → give 2-3 line explanation
- If asking for list → give clean list only

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 80
                }
            }
        )

        return response.json()["response"]

    except:
        return "⚠️ Error: Ollama not running"

# -------------------- SESSION --------------------
if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- LOAD SAVED DATA --------------------
if os.path.exists("storage/index.faiss"):
    index = faiss.read_index("storage/index.faiss")

    with open("storage/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    st.session_state.index = index
    st.session_state.chunks = chunks

# -------------------- SIDEBAR --------------------
st.sidebar.title("📂 Upload Notes")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if st.sidebar.button("🗑 Clear Saved Data"):
    if os.path.exists("storage/index.faiss"):
        os.remove("storage/index.faiss")
        os.remove("storage/chunks.pkl")
        st.sidebar.warning("Data cleared. Refresh page.")

# -------------------- TITLE --------------------
st.title("📘 AI Study Assistant (LLM Powered)")

# -------------------- PROCESS FILES --------------------
if uploaded_files:
    all_text = ""

    for file in uploaded_files:
        all_text += read_pdf(file)

    chunks = chunk_text(all_text)

    with st.spinner("Processing..."):
        embeddings = model.encode(chunks)
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

    st.session_state.index = index
    st.session_state.chunks = chunks

    # Save
    if not os.path.exists("storage"):
        os.makedirs("storage")

    faiss.write_index(index, "storage/index.faiss")

    with open("storage/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    st.sidebar.success("✅ Notes processed & saved")

# -------------------- DISPLAY CHAT --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- CHAT INPUT --------------------
query = st.chat_input("Ask something...")

if query:
    if st.session_state.index is None:
        st.warning("Upload PDFs first")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            query_vec = model.encode([query])
            D, I = st.session_state.index.search(np.array(query_vec), k=3)

            # LIMIT context (important for LLM)
            retrieved_text = " ".join([st.session_state.chunks[i] for i in I[0][:2]])

            answer = ask_llm(retrieved_text, query)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

# -------------------- EMPTY STATE --------------------
if st.session_state.index is None:
    st.info("⬅️ Upload PDFs to start")