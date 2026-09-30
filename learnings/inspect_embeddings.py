import ollama
import numpy as np

def consine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    return dot_product / (norm_A * norm_B)


texts = [
    "The cat is sleeping on the sofa.",
    "A kitten is resting on the couch.",
    "The dog is running in the garden.",
    "Python is a popular programming language.",
    "Java is commonly used for backend development.",
    "The weather is sunny today.",
]

embeddings = ollama.embed(
                model="nomic-embed-text:latest ",
                input=texts
                )["embeddings"]

embeddings = np.array(embeddings)

for i, embedding_a in enumerate(embeddings):
    for j, embedding_b in enumerate(embeddings):
        if j > i:
            print(f"Cosine Similarity[{i}, {j}]: ", consine_similarity(embedding_a, embedding_b))
    print()
