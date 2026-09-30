evaluation_data = [
    {
        "question": "How many days do employees have to submit an expense claim?",
        "retrieved": ["chunk_103", "chunk_101", "chunk_104"],
        "relevant": ["chunk_101"]
    },
    {
        "question": "Do expense claims require receipts?",
        "retrieved": ["chunk_104", "chunk_101", "chunk_108"],
        "relevant": ["chunk_104"]
    },
    {
        "question": "How many annual leave days do employees receive?",
        "retrieved": ["chunk_101", "chunk_103", "chunk_107"],
        "relevant": ["chunk_103"]
    }
]

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(
        chunk in relevant
        for chunk in retrieved_k
    )

    return relevant_retrieved / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(
        chunk in relevant
        for chunk in retrieved_k
    )

    return relevant_retrieved / len(relevant)

def reciprocal_rank(retrieved, relevant):
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk in relevant:
            return 1 / rank

    return 0

def hit_rate_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    for chunk in retrieved_k:
        if chunk in relevant:
            return 1
    return 0

retrieved = [
    "chunk_103",
    "chunk_101",
    "chunk_104"
]

relevant = [
    "chunk_101"
]
precisions = []
recalls = []
rr = []
hit_rates = []
for data in evaluation_data:
    retrieved = data["retrieved"]
    relevant = data["relevant"]
    precisions.append(precision_at_k(retrieved, relevant, 3))
    recalls.append(recall_at_k(retrieved, relevant, 3))
    rr.append(reciprocal_rank(retrieved, relevant))
    hit_rates.append(hit_rate_at_k(retrieved, relevant, 3))

print("Average Precision@3", sum(precisions) / len(precisions))
print("Average Recall@3", sum(recalls) / len(recalls))
print("Average MRR", sum(rr) / len(rr))
print("Average Hit Rate@3", sum(hit_rates) / len(hit_rates))