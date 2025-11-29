import streamlit as st
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
import RAGPipeline

DB_PATH = 'edu_chunks.db'
IDX_PATH = 'edu_index.faiss'
MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL = 'mistral'
TOP_K_DEFAULT = 3

@st.cache_resource
def load_resources():
    embedder = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(IDX_PATH)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return embedder, index, conn

embedder, index, conn = load_resources()

st.set_page_config(page_title="EduLLM Tutor", layout="wide")
st.title("🎓 EduLLM — Offline AI Tutor")
st.caption("RAG-based Intelligent Tutoring System (Class 11–12 Science)")

st.sidebar.header("Settings")
TOP_K = st.sidebar.slider("Number of retrieved chunks", 1, 10, TOP_K_DEFAULT)
show_context = st.sidebar.checkbox("Show retrieved context", True)

query = st.text_area("Ask a question:", placeholder="e.g., Explain Figure 2.3 in Physics or What is dimensional analysis?")

if st.button("Generate Answer"):
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            context = ''
            title_match = RAGPipeline.detect_direct_ref(query)
            retrieved_chunks = {}

            if title_match:
                retrieved_chunks = RAGPipeline.retrieve_by_title(title_match, conn)
                if retrieved_chunks:
                    context = "\n".join(chunk['content'] for chunk in retrieved_chunks.values())
                    for chunk in retrieved_chunks.values():
                        if chunk['parent_section_id']:
                            parent = RAGPipeline.retrieve_by_id(chunk['parent_section_id'], conn)
                            if parent:
                                context += "\n" + "\n".join(parent.values())
                else:
                    ids = RAGPipeline.search_faiss(index, embedder, query, TOP_K)
                    retrieved_chunks = RAGPipeline.retrieve_similar_chunks(conn, ids)
            else:
                ids = RAGPipeline.search_faiss(index, embedder, query, TOP_K)
                retrieved_chunks = RAGPipeline.retrieve_similar_chunks(conn, ids)

            if retrieved_chunks:
                context += "\n\n".join(chunk['content'] for chunk in retrieved_chunks.values())
                for chunk in retrieved_chunks.values():
                    if chunk['parent_section_id']:
                        parent = RAGPipeline.retrieve_by_id(chunk['parent_section_id'], conn)
                        if parent:
                            context += "\n" + "\n".join(parent.values())

            if context:
                full_prompt = f"Get context for the question from the given text. Don't limit yourself to the text:\n\n{context}\n\nQuestion: {query}"
            else:
                full_prompt = f"Answer this question: {query}"

            answer = RAGPipeline.run_ollama(full_prompt, LLM_MODEL)

        st.subheader("Answer")
        st.write(answer)

        if show_context and context:
            st.subheader("Retrieved Context")
            st.text_area("Context", context, height=300)

st.markdown("---")
st.caption("Powered by local FAISS, SQLite, and Ollama for fully offline learning.")
