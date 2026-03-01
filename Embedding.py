import json
import numpy as np
from sentence_transformers import SentenceTransformer

input_file = r'Chapters\Physics\Class 11\Chap2_Chunked.json'
embedding_output_file = r'Chapters\Physics\Class 11\Chap2_Embeddings.json'

with open(input_file, 'r', encoding='utf-8') as file:
    sections = json.load(file)

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

embedding_list = []
for chunk in sections:
    text = chunk.get('content', '').strip()
    if not text:
        continue
    embedding = sentence_transformer.encode(text)
    embedding_list.append({
        'id': chunk.get('id'),
        'type': chunk.get('type'),
        'class': chunk.get('class'),
        'subject': chunk.get('subject'),
        'chapter': chunk.get('chapter'),
        'section_title': chunk.get('section_title'),
        'section_id': chunk.get('section_id'),
        'parent_section_id': chunk.get('parent_section_id'),
        'title': chunk.get('title'),
        'content': text,
        'chunk_no': chunk.get('chunk_no'),
        'total_chunks': chunk.get('total_chunks'),
        'embedding': embedding.tolist()
    })

with open(embedding_output_file, 'w', encoding='utf-8') as file:
    json.dump(embedding_list, file, ensure_ascii=False, indent=4)
