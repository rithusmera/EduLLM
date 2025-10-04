import numpy as np
import faiss
import sqlite3
import json

with open(r'Chapters\Physics\Class 11\Chap1_Chunked.json', 'r', encoding= 'utf-8') as file:
    text_chunks= json.load(file)

with open(r'Chapters\Physics\Class 11\Chap1_Embeddings.json', 'r', encoding='utf-8') as file:
    embedding_data= json.load(file)

chunks_map = {}
for t in text_chunks:
    key= t['Title'], t['Chunk No.'], t['Section ID']
    chunks_map[key]= t['Body']

conn = sqlite3.connect('edu_chunks.db')
cursor= conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS chunks(
                    chunk_id INTEGER PRIMARY KEY,
                    title TEXT,
                    chunk_no INTEGER,
                    total_chunks INTEGER,
                    section_id TEXT,
                    text TEXT
                )''')

conn.commit()

embedding_dim = len(embedding_data[0]['Embedding'])
index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))

chunk_id_counter = 0
for e in embedding_data:
    key = e['Title'], e['Chunk No.'], e['Section ID']
    body = chunks_map.get(key, '').strip()

    if not body:
        continue

    embedding = np.array(e['Embedding'], dtype='float32').reshape(1, -1)
    index.add_with_ids(embedding, np.array([chunk_id_counter], dtype='int64'))

    cursor.execute('''INSERT OR REPLACE INTO 
                   chunks(chunk_id, title, chunk_no, total_chunks, section_id, text)
                   VALUES(?,?,?,?,?,?)
                   ''',(
                       chunk_id_counter,
                       e['Title'],
                       e['Chunk No.'],
                       e['Total Chunks'],
                       e['Section ID'],
                       body
                   ))
    
    chunk_id_counter+=1

conn.commit()
conn.close()

faiss.write_index(index, "edu_index.faiss")

print(f"FAISS + SQLite setup complete with {chunk_id_counter} chunks.")

    
