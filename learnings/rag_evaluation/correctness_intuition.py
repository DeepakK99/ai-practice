from enum import Enum
from pydantic import BaseModel
from ollama import chat


class Correctness(str, Enum):
    CORRECT = "C"
    PARTIALLY_CORRECT = "P"
    INCORRECT = "I"


class CorrectnessEvaluation(BaseModel):
    correctness: Correctness
    explanation: str


# example 
question = """
How many days do employees have to submit an expense claim?
"""

ground_truth_answer = """
Employees must submit expense claims within 30 days.
Receipts are required for expenses above $25.
"""

generated_answer = """
Employees need to submit expense claims within the required period.
"""

# building the judge
prompt = f"""
You are an expert evaluator assessing the correctness of a generated answer.

Compare the GENERATED ANSWER against the GROUND TRUTH ANSWER
while considering the QUESTION.

Evaluation criteria:

C (Correct):
The generated answer provides the correct core information needed to
answer the question and does not contradict the ground truth.

P (Partially Correct):
The generated answer contains some correct information but is incomplete,
misses an important part of the ground truth, or contains a minor factual
error while still addressing part of the question.

I (Incorrect):
The generated answer is factually wrong, contradicts the ground truth,
or fails to answer the question.

Rules:
- Focus on semantic meaning, not exact wording.
- Ignore differences in wording, punctuation, formatting, and verbosity.
- Do not require the generated answer to use the same wording as the ground truth.
- Additional information does not make an answer incorrect unless that
  information contradicts the ground truth.
- Base the evaluation only on the question, ground truth answer, and
  generated answer.
- Do not use outside knowledge to change the evaluation.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth_answer}

GENERATED ANSWER:
{generated_answer}
"""

response = chat(model="lukaspetrik/gemma3-tools:4b",
                messages=[{"role": "user", "content": prompt}],
                format=CorrectnessEvaluation.model_json_schema()
                )
eval = CorrectnessEvaluation.model_validate_json(response.message.content)
print(eval.correctness)
