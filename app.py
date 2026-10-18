"""
PDF Reader RAG - Streamlit + FAISS + Google Gemini
- Default: repo mein rakhi PDF (engro_annual_report_2024.pdf)
- Ya sidebar se koi bhi apni PDF upload karo
- Index app khud banati hai (alag index.faiss ki zarurat nahi)
"""
import io
import os

import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from google import genai
from google.genai import types
from pypdf import PdfReader

try:
    import faiss
except ImportError:  # faiss install na ho to numpy se search chalega
    faiss = None

DEFAULT_PDF = "engro_annual_report_2024.pdf"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL = "gemini-flash-latest"  # hamesha current Flash model (free tier)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

st.set_page_config(page_title="PDF Reader RAG", page_icon="📄")
st.title("📄 PDF Reader RAG")


@st.cache_resource
def get_embedder():
    return TextEmbedding(EMBED_MODEL)


def split_text(text, size, overlap):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


@st.cache_resource(show_spinner=False)
def build_index(pdf_bytes: bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < 50:
            continue
        for piece in split_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            if piece.strip():
                chunks.append({"page": page_no, "text": piece})

    if not chunks:
        return None, [], len(reader.pages)

    texts = [c["text"] for c in chunks]
    emb = get_embedder()
    parts = []
    for i in range(0, len(texts), 64):  # chhoti batches = kam memory
        batch = texts[i:i + 64]
        parts.append(np.array(list(emb.embed(batch, batch_size=16)), dtype="float32"))
    vecs = np.vstack(parts)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    if faiss is not None:
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
    else:
        index = vecs  # numpy fallback
    return index, chunks, len(reader.pages)


def get_client():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    key = key or os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=key) if key else None


# ---------- PDF choose karo ----------
st.sidebar.header("Document")
choice = st.sidebar.radio("PDF source", ["Default (Engro Annual Report 2024)", "Upload my own PDF"])

pdf_bytes, doc_name = None, ""
if choice.startswith("Default"):
    if os.path.exists(DEFAULT_PDF):
        with open(DEFAULT_PDF, "rb") as f:
            pdf_bytes = f.read()
        doc_name = DEFAULT_PDF
    else:
        st.sidebar.error(f"{DEFAULT_PDF} repo mein nahi mili.")
else:
    up = st.sidebar.file_uploader("PDF upload karein", type="pdf")
    if up:
        pdf_bytes, doc_name = up.getvalue(), up.name

if not pdf_bytes:
    st.info("Pehle koi PDF select ya upload karein.")
    st.stop()

with st.spinner("PDF padh raha hoon aur index bana raha hoon (pehli baar thora time lagta hai)..."):
    index, chunks, n_pages = build_index(pdf_bytes)

if index is None:
    st.error("Is PDF se text nahi nikla (shayad scanned/image PDF hai).")
    st.stop()

st.caption(f"📎 {doc_name} • {n_pages} pages • {len(chunks)} chunks • search: {'FAISS' if faiss is not None else 'numpy (faiss not installed)'}")

# ---------- Sawal-jawab ----------
top_k = st.sidebar.slider("Retrieved chunks", 2, 10, 5)
question = st.text_input("Is PDF ke baare mein sawal likhein:")

if question:
    client = get_client()
    if client is None:
        st.error("GEMINI_API_KEY set nahi hai. Streamlit app Settings > Secrets mein add karein.")
        st.stop()

    q_vec = np.array(list(get_embedder().embed([question])), dtype="float32")
    q_vec /= np.linalg.norm(q_vec, axis=1, keepdims=True)
    if faiss is not None:
        scores, ids = index.search(q_vec, top_k)
    else:
        sims = index @ q_vec[0]
        top = np.argsort(-sims)[:top_k]
        scores, ids = np.array([sims[top]]), np.array([top])
    hits = [(float(s), chunks[i]) for s, i in zip(scores[0], ids[0])]

    context = "\n\n".join(f"[Page {h['page']}]\n{h['text']}" for _, h in hits)
    system_text = (
        "You answer questions strictly from the provided PDF context. "
        "Mention page numbers. If the answer is not in the context, say you could not find it. "
        "Reply in the same language as the question."
    )
    try:
        with st.spinner("Jawab bana raha hoon..."):
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"Context:\n{context}\n\nQuestion: {question}",
                config=types.GenerateContentConfig(system_instruction=system_text, max_output_tokens=1500),
            )
        st.subheader("Answer")
        st.write(resp.text)
    except Exception as e:
        st.warning(f"Gemini se jawab nahi bana ({type(e).__name__}): {str(e)[:200]}")

    st.subheader("Sources")
    for score, h in hits:
        with st.expander(f"Page {h['page']} (score {score:.3f})"):
            st.write(h["text"])
