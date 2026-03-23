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

# -------------------- HUGGINGFACE CONFIG --------------------
API_URL =  "https://api-inference.huggingface.co/models/google/flan-t5-base"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
    "Content-Type": "application/json"
}

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


def ask_llm(context, question):
    prompt = f"""
You are a helpful study assistant.

Rules:
- If greeting → reply normally
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
            API_URL,
            headers=HEADERS,
            json={"inputs": prompt}
        )

        if response.status_code == 200:
            result = response.json()
            return result[0]["generated_text"]

        elif response.status_code == 503:
            return "⏳ Model loading... try again in few seconds."

        elif response.status_code == 401:
            return "❌ Invalid API token."

        else:
            return f"⚠️ API Error: {response.text}"

    except Exception as e:
        return f"⚠️ Exception: {str(e)}"


# -------------------- SESSION --------------------
if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- LOAD SAVED --------------------
if os.path.exists("storage/index.faiss"):
    index = faiss.read_index("storage/index.faiss")
    with open("storage/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    st.session_state.index = index
    st.session_state.chunks = chunks

# -------------------- SIDEBAR --------------------
st.sidebar.title("📂 Upload PDFs")

files = st.sidebar.file_uploader("Upload", type="pdf", accept_multiple_files=True)

if st.sidebar.button("🗑 Clear Data"):
    if os.path.exists("storage/index.faiss"):
        os.remove("storage/index.faiss")
        os.remove("storage/chunks.pkl")
        st.sidebar.warning("Cleared!")

# -------------------- TITLE --------------------
st.title("📘 AI Study Assistant (Live)")

# -------------------- PROCESS --------------------
if files:
    full_text = ""

    for file in files:
        full_text += read_pdf(file)

    chunks = chunk_text(full_text)

    with st.spinner("Processing..."):
        embeddings = model.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))

    st.session_state.index = index
    st.session_state.chunks = chunks

    os.makedirs("storage", exist_ok=True)
    faiss.write_index(index, "storage/index.faiss")

    with open("storage/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    st.sidebar.success("✅ Ready!")

# -------------------- CHAT --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask from your notes...")

if query:
    if st.session_state.index is None:
        st.warning("Upload PDFs first")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            vec = model.encode([query])
            D, I = st.session_state.index.search(np.array(vec), k=2)

            context = " ".join([st.session_state.chunks[i] for i in I[0]])

            answer = ask_llm(context, query)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

# -------------------- EMPTY --------------------
if st.session_state.index is None:
    st.info("⬅️ Upload PDFs to begin")