import os
import json
import numpy as np
import faiss
import sqlite3

CHUNKS_PATH = r'Chapters\Physics\Class 11\Chap2_Chunked.json'
EMB_PATH = r'Chapters\Physics\Class 11\Chap2_Embeddings.json'
DB_PATH = 'edu_chunks.db'
INDEX_PATH = 'edu_index.faiss'

with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

with open(EMB_PATH, 'r', encoding='utf-8') as f:
    embeddings = json.load(f)

chunk_map = {c['id']: c for c in chunks}

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS chunks(
                    chunk_id INTEGER PRIMARY KEY,
                    id TEXT,
                    type TEXT,
                    class TEXT,
                    subject TEXT,
                    chapter TEXT,
                    section_title TEXT,
                    section_id TEXT,
                    parent_section_id TEXT,
                    title TEXT,
                    content TEXT,
                    chunk_no INTEGER,
                    total_chunks INTEGER
                )''')
conn.commit()

embedding_dim = len(embeddings[0]['embedding'])

if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))

cursor.execute("SELECT IFNULL(MAX(chunk_id), -1) FROM chunks")
counter = cursor.fetchone()[0] + 1

for e in embeddings:
    emb_id = e['id']
    matched_chunk = chunk_map.get(emb_id)

    if not matched_chunk:
        continue

    embedding = np.array(e['embedding'], dtype='float32').reshape(1, -1)
    embedding /= np.linalg.norm(embedding) 
    index.add_with_ids(embedding, np.array([counter], dtype='int64'))

    cursor.execute('''INSERT OR REPLACE INTO chunks(
                        chunk_id, id, type, class, subject, chapter, section_title, section_id,
                        parent_section_id, title, content, chunk_no, total_chunks)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (counter,
                    matched_chunk['id'],
                    matched_chunk['type'],
                    matched_chunk['class'],
                    matched_chunk['subject'],
                    matched_chunk['chapter'],
                    matched_chunk['section_title'],
                    matched_chunk['section_id'],
                    matched_chunk['parent_section_id'],
                    matched_chunk['title'],
                    matched_chunk['content'],
                    matched_chunk['chunk_no'],
                    matched_chunk['total_chunks'])
                   )

    counter += 1

conn.commit()
conn.close()

faiss.write_index(index, INDEX_PATH)
print(f"Added {len(embeddings)} new chunks. Total now: {counter}")

