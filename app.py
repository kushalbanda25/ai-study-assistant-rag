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
st.set_page_config(page_title="AI Study Assistant", layout="centered")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------- OPENROUTER FUNCTION --------------------
def ask_llm(context, question):
    prompt = f"""
You are a helpful study assistant.

Rules:
- If greeting → respond normally
- Else → answer ONLY from context
- Use bullet points
- Keep answers short and clear

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    },
    json={
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
)

        result = response.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return f"⚠️ API Error: {result}"

    except Exception as e:
        return f"⚠️ Exception: {str(e)}"


# -------------------- FUNCTIONS --------------------
def read_pdf(file):
    text = ""
    pdf = PdfReader(file)

    for page in pdf.pages:
        content = page.extract_text()
        if content:
            text += content + " "

    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text, chunk_size=100):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


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

if st.sidebar.button("🗑 Clear Data"):
    if os.path.exists("storage/index.faiss"):
        os.remove("storage/index.faiss")
        os.remove("storage/chunks.pkl")
        st.sidebar.warning("Data cleared. Refresh page.")


# -------------------- TITLE --------------------
st.title("📘 AI Study Assistant (Live GPT Version)")

# -------------------- PROCESS FILES --------------------
if uploaded_files:
    all_text = ""

    for file in uploaded_files:
        all_text += read_pdf(file)

    chunks = chunk_text(all_text)

    with st.spinner("Processing notes..."):
        embeddings = model.encode(chunks)
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

    st.session_state.index = index
    st.session_state.chunks = chunks

    os.makedirs("storage", exist_ok=True)
    faiss.write_index(index, "storage/index.faiss")

    with open("storage/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    st.sidebar.success("✅ Notes processed & saved")


# -------------------- CHAT DISPLAY --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------- CHAT INPUT --------------------
query = st.chat_input("Ask something from your notes...")

if query:
    if st.session_state.index is None:
        st.warning("Upload PDFs first")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            query_vec = model.encode([query])
            D, I = st.session_state.index.search(np.array(query_vec), k=2)

            context = " ".join([st.session_state.chunks[i] for i in I[0]])

            answer = ask_llm(context, query)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)


# -------------------- EMPTY STATE --------------------
if st.session_state.index is None:
    st.info("⬅️ Upload PDFs to begin")