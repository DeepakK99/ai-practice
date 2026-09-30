def evaluate_rag(
    question,
    retrieved_chunks,
    relevant_chunks,
    relevance_scores,
    ground_truth_answer,
    generated_answer,
    k=3,
):

    return {
        # Retrieval evaluation
        "precision": precision(
            retrieved_chunks,
            relevant_chunks,
            k
        ),

        "recall": recall(
            retrieved_chunks,
            relevant_chunks,
            k
        ),

        "mrr": mrr(
            retrieved_chunks,
            relevant_chunks
        ),

        "hit_rate": hit_rate(
            retrieved_chunks,
            relevant_chunks,
            k
        ),

        "ndcg": ndcg(
            retrieved_chunks,
            relevance_scores,
            k
        ),

        # Generation evaluation
        "faithfulness": faithfulness(
            question,
            retrieved_chunks,
            generated_answer
        ),

        "correctness": correctness(
            question,
            ground_truth_answer,
            generated_answer
        ),

        "context_relevance": context_relevance(
            question,
            retrieved_chunks
        )
    }

BETTER_OUTPUT = {
    "retrieval": {
        "precision@3": ...,
        "recall@3": ...,
        "mrr": ...,
        "hit_rate@3": ...,
        "ndcg@3": ...
    },

    "generation": {
        "faithfulness": ...,
        "correctness": ...,
        "context_relevance": ...
    },

    "system": {
        "latency_ms": ...,
        "input_tokens": ...,
        "output_tokens": ...
    }
}