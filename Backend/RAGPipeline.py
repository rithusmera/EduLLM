import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
import subprocess

DB_PATH = 'edu_chunks.db'
IDX_PATH = 'edu_index.faiss'
TOP_K = 3
MODEL = 'mistral'

def load_index(IDX_PATH):
    return faiss.read_index(IDX_PATH)

def search_faiss(index, query_vec, k):
    _ , indices = index.search(query_vec, k)
    return indices[0]

def retrieve_text(conn, ids):
    cursor = conn.cursor()
    placeholder = ','.join('?' * len(ids))
    query= f'SELECT chunk_id, content FROM chunks WHERE chunk_id IN ({placeholder})'
    cursor.execute(query, ids)
    return({row[0]: row[1] for row in cursor.fetchall()})

def run_ollama(prompt, model):
    cmd = ['ollama', 'run', model]
    process = subprocess.Popen(cmd, stdin= subprocess.PIPE, stdout= subprocess.PIPE, text=True)
    stdout, _ = process.communicate(prompt)
    return stdout

def main():

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    index = load_index(IDX_PATH)
    conn = sqlite3.connect(DB_PATH)

    print("EduLLM RAG Pipeline (type 'exit' to quit)")

    while True:

        query = input('Enter a question: ')

        if query.lower() == 'exit':
            break

        query_vector = embedder.encode([query]).astype('float32').reshape(1, -1)

        ids = search_faiss(index, query_vector, TOP_K)
        texts = retrieve_text(conn, ids)

        if not texts:
            full_prompt = f'Answer this question on a high school level. Question: {query}'
        else: 
            context = '\n\n'.join(texts.values())
            full_prompt =  f"Get context for the question from the given text.  Dont limit yourself to the text :\n\n{context}\n\nQuestion: {query}"

        print('Model Response: \n')
        answer = run_ollama(full_prompt, MODEL)
        print(answer)

    conn.close()
    print('Exiting Program')

if __name__ == '__main__':
    main()

