import json
import numpy as np
from sentence_transformers import SentenceTransformer

input_file = r'.\Chunks\Sample_chunks.json'
embedding_output_file = r'.\Embeddings\Sample_embeddings.json'

with open(input_file, 'r', encoding='utf-8') as file:
    sections = json.load(file)

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

embedding_list = []
for chunk in sections:
    text = chunk['Body']
    embedding = sentence_transformer.encode(text)
    embedding_list.append({
        'Title': chunk['Title'],
        'Chunk No.': chunk['Chunk No.'],
        'Total Chunks': chunk['Total Chunks'],
        'Section ID': chunk['Section ID'],
        'Body': text,
        'Embedding': embedding.tolist()
    })

with open(embedding_output_file, 'w', encoding='utf-8') as file:
    json.dump(embedding_list, file, ensure_ascii=False, indent=4)
