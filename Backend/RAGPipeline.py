import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
import subprocess
import re
import numpy as np

DB_PATH = 'edu_chunks.db'
IDX_PATH = 'edu_index.faiss'
TOP_K = 3
MODEL = 'mistral'

def load_index(IDX_PATH):
    return faiss.read_index(IDX_PATH)

def search_faiss(index, query_vec, k):
    '''Similarity search in faiss index'''

    _ , indices = index.search(query_vec, k)
    return indices[0]

def retrieve_similar_chunks(conn, ids):
    '''Retrieve chunks of ids obtained from faiss search'''

    cursor = conn.cursor()

    ids = [int(i) for i in ids if i != -1]  
    if not ids:
        return {}

    placeholder = ','.join('?' * len(ids))
    query= f'SELECT chunk_id, content, parent_section_id FROM chunks WHERE chunk_id IN ({placeholder})'
    cursor.execute(query, ids)
    return({row[0]: {'content': row[1], 'parent_section_id': row[2]} for row in cursor.fetchall()})

def retrieve_by_title(title, conn):
    '''Retrieving tables/figures/examples/exercises directly through title match instead of faiss similarity seach'''

    cursor = conn.cursor()
    cursor.execute('SELECT id, content, parent_section_id FROM chunks WHERE LOWER(title) LIKE ?', (f"%{title.lower()}%", ))
    rows = cursor.fetchall()
    if rows:
        return({row[0]: {'content': row[1], 'parent_section_id': row[2]} for row in rows}) 
    return None
    
def retrieve_by_id(id, conn):
    '''Retrieving parent section of table/figure/example/exercise retrieved through title match'''

    cursor = conn.cursor()
    cursor.execute('SELECT id, content FROM chunks WHERE id = ?', (id, ))
    row = cursor.fetchone()
    if row:
        return {row[0]: row[1]}
    return None

def run_ollama(prompt, model):
    cmd = ['ollama', 'run', model]
    process = subprocess.Popen(cmd, stdin= subprocess.PIPE, stdout= subprocess.PIPE, text=True)
    stdout, _ = process.communicate(prompt)
    return stdout

def detect_direct_ref(query):
    '''Direct similarity check from query for words like table, figure, example or exercise'''

    pattern = r'\b(figure|table|example|exercise)\s+(\d+(\.\d+)*)\b'
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        return f'{match.group(1).lower()} {match.group(2)}'
    return None

def main():

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    index = load_index(IDX_PATH)
    conn = sqlite3.connect(DB_PATH)

    print("EduLLM RAG Pipeline (type 'exit' to quit)")

    while True:

        query = input('Enter a question: ')
        if query.lower() == 'exit':
            break

        context = ''

        title_match = detect_direct_ref(query)

        if title_match:
            print(f"Title match found: {title_match}\n")

            chunk_match = retrieve_by_title(title_match, conn)
            if  chunk_match:
                print(f"Chunk match found: {chunk_match.keys()}")

                context = "\n".join(chunk.get('content','') for chunk in chunk_match.values())

                for chunk in chunk_match.values():
                    if chunk['parent_section_id']:
                        print(f'Chunk has parent section: {chunk['parent_section_id']}')
                        parent_section = retrieve_by_id(chunk['parent_section_id'], conn)

                        if parent_section:
                            print(f"Parent section: {parent_section.keys()}")
                            context+= "\n".join(parent_section.values())
                
            else:
                query_vector = embedder.encode([query]).astype('float32').reshape(1, -1)
                query_vector /= np.linalg.norm(query_vector, axis=1, keepdims=True)

                ids = search_faiss(index, query_vector, TOP_K)
                texts = retrieve_similar_chunks(conn, ids)

                if texts: 
                    print(f"Texts retreived via faiss similarity search: {texts.keys()}")

                    context = '\n\n'.join(chunk['content'] for chunk in texts.values())

                    for chunk in texts.values():
                        if chunk['parent_section_id']:
                            print(f'Parent section found: {chunk['parent_section_id']}')
                            parent_section = retrieve_by_id(chunk['parent_section_id'], conn)

                            if parent_section:
                                print(f"Parent section: {parent_section.keys()}")
                                context+= "\n".join(parent_section.values())                    

        else:    
            print('Title match now found')                                         
            query_vector = embedder.encode([query]).astype('float32').reshape(1, -1)
            query_vector /= np.linalg.norm(query_vector, axis=1, keepdims=True)

            ids = search_faiss(index, query_vector, TOP_K)
            print(f'Ids retrieved:{ids}')
            texts = retrieve_similar_chunks(conn, ids)
            print(f"Texts retreived via faiss similarity search: {texts.keys()}")

            if texts: 
                print(f"Texts retreived via faiss similarity search: {texts.keys()}")

                context = '\n\n'.join(chunk['content'] for chunk in texts.values())

                for chunk in texts.values():
                    if chunk['parent_section_id']:
                        parent_section = retrieve_by_id(chunk['parent_section_id'], conn)

                        if parent_section:
                            context+= "\n".join(parent_section.values())

        if context:
            full_prompt =  f"Get context for the question from the given text.Dont limit yourself to the text :\n\n{context}\n\nQuestion: {query}"    
        else:
            full_prompt = f'Answer this question: {query}'

        print(f'\n\n{full_prompt}')
        print('Model Response: \n')
        answer = run_ollama(full_prompt, MODEL)
        print(answer)

    conn.close()
    print('Exiting Program')

if __name__ == '__main__':
    main()

