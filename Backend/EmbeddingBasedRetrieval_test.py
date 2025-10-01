import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(r'.\Embeddings\Sample_embeddings.json','r',encoding='utf-8') as file:
    chunks = json.load(file)

embeddings = []

for chunk in chunks:
    embeddings.append(chunk['Embedding'])

embeddings = np.array(embeddings)

while True:
    question = input("Enter a question: ")

    if question.lower() == 'exit':
        break

    question_embedding = model.encode(question)
    question_embedding = question_embedding.reshape(1, -1)

    similarity = cosine_similarity(question_embedding, embeddings)[0]

    top_indices = similarity.argsort()[::-1][:3]

    print("\nTop matching sections:")
    for idx in top_indices:
        score = similarity[idx]
        section = chunks[idx]
        print(f"\n🔹 Title: {section['Title']} (Chunk {section['Chunk No.']}/{section['Total Chunks']})")
        print(f"   Section ID: {section['Section ID']}")
        print(f"   Similarity Score: {score:.4f}")
        print(f"   Content Preview:\n{section['Body'][:500]}...")




