from pydantic import BaseModel
from ollama import chat


class ClaimEvaluation(BaseModel):
    claim: str
    supported: bool


class FaithfulnessEvaluation(BaseModel):
    claims: list[ClaimEvaluation]


# example 
question = """
How many days do employees have to submit an expense claim?
"""

context = """
Employees must submit expense claims within 30 days.
Receipts are required for expenses above $25.
"""

answer = """
Employees have 30 days to submit an expense claim.
Late claims are automatically rejected.
"""

# building the judge
prompt = f"""
You are evaluating the faithfulness of a RAG answer.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Your task:

1. Extract the factual claims made in the answer.
2. For each claim, determine whether it is fully supported by the provided context.
3. A claim is supported only if the context provides sufficient evidence for it.
4. Do not use outside knowledge.
5. Mark a claim as unsupported if the context does not provide evidence for it.

Return one evaluation for every factual claim.
"""

response = chat(model="lukaspetrik/gemma3-tools:4b",
                messages=[{"role": "user", "content": prompt}],
                format=FaithfulnessEvaluation.model_json_schema()
                )

eval = FaithfulnessEvaluation.model_validate_json(response.message.content)
print(eval)
supported = 0
total = len(eval.claims)
for claim in eval.claims:
    if claim.supported:
        supported += 1
print("Faithfulness: ", round((supported/total), 2))

