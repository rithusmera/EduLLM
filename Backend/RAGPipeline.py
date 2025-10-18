import faiss
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess
import json

# ======== CONFIG ==========
DB_PATH = "edu_chunks.db"
INDEX_PATH = "edu_index.faiss"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
TOP_K = 3
# ==========================


def load_faiss_index(index_path):
    """Load FAISS index from file"""
    return faiss.read_index(index_path)


def search_faiss(index, query_embedding, k=3):
    """Search for top-k similar vectors"""
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]


def get_text_from_sqlite(conn, ids):
    """Retrieve text chunks from SQLite by ID"""
    cursor = conn.cursor()
    placeholders = ",".join("?" * len(ids))
    query = f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})"
    cursor.execute(query, ids)
    return {row[0]: row[1] for row in cursor.fetchall()}


def query_ollama(model_name, prompt):
    """Send a prompt to Ollama and return plain text output"""
    cmd = ["ollama", "run", model_name]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    stdout, _ = process.communicate(prompt)
    return stdout


def main():
    # Load embedding model
    embedder = SentenceTransformer(MODEL_NAME)

    # Load FAISS + SQLite
    index = load_faiss_index(INDEX_PATH)
    conn = sqlite3.connect(DB_PATH)

    # User query
    query = input("Enter your question: ")

    # Create query embedding
    query_vector = embedder.encode([query]).astype("float32").reshape(1, -1)

    # Retrieve top-k chunks
    ids, distances = search_faiss(index, query_vector, TOP_K)
    texts = get_text_from_sqlite(conn, [int(i) for i in ids])

    if not texts:
        print("No matching chunks found.")
        return

    # Combine retrieved context
    context = "\n\n".join(texts.values())
    full_prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    # Query Ollama
    print("\n--- Model Response ---\n")
    answer = query_ollama(OLLAMA_MODEL, full_prompt)
    print(answer)

    conn.close()


if __name__ == "__main__":
    main()


