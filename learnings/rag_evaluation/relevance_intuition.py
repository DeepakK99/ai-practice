from enum import Enum
from pydantic import BaseModel
from ollama import chat

class RelevanceEvaluation(BaseModel):
    relevance: int
    explanation: str


# example 
question = """
How many days do employees have to submit an expense claim?
"""

context = """
Employees must submit expense claims within 30 days.
"""

# building the judge
prompt = f"""
You are an expert evaluator assessing the relevance of retrieved context
for a question in a RAG system.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Evaluate how relevant the retrieved context is for answering the question.

Scoring:
2 = Highly relevant
The context contains information directly useful for answering the question.

1 = Partially relevant
The context contains some useful information but also misses important
information or contains substantial irrelevant information.

0 = Irrelevant
The context does not contain useful information for answering the question.

Rules:
- Judge relevance to the question, not whether the context is factually true.
- Do not use outside knowledge.
- Focus on whether the context would help an LLM answer the question.

Return the score and a brief explanation.
"""

response = chat(model="lukaspetrik/gemma3-tools:4b",
                messages=[{"role": "user", "content": prompt}],
                format=RelevanceEvaluation.model_json_schema()
                )
eval = RelevanceEvaluation.model_validate_json(response.message.content)
print(eval.relevance)
