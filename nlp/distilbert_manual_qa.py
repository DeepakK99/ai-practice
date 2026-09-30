import torch

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)


MODEL_NAME = "distilbert-base-uncased-distilled-squad"


# ---------------------------------------------------------
# 1. LOAD TOKENIZER
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------
# 2. LOAD QA MODEL
# ---------------------------------------------------------

model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_NAME
)


# ---------------------------------------------------------
# 3. QUESTION + CONTEXT
# ---------------------------------------------------------

question = "Where is Amazon headquartered?"

context = """
Amazon is headquartered in Seattle, Washington.
The company was founded by Jeff Bezos in 1994.
"""


# ---------------------------------------------------------
# 4. TOKENIZE QUESTION + CONTEXT
# ---------------------------------------------------------

inputs = tokenizer(
    question,
    context,
    return_tensors="pt",
)

print("Inputs: ",
    tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )
)
# ---------------------------------------------------------
# 5. RUN MODEL
# ---------------------------------------------------------

with torch.no_grad():
    outputs = model(**inputs)


# ---------------------------------------------------------
# 6. GET START / END LOGITS
# ---------------------------------------------------------

start_logits = outputs.start_logits
end_logits = outputs.end_logits


print("START LOGITS:")
print(start_logits)

print("\nEND LOGITS:")
print(end_logits)


# ---------------------------------------------------------
# 7. FIND BEST START / END POSITIONS
# ---------------------------------------------------------

start_position = torch.argmax(
    start_logits,
    dim=-1,
).item()

end_position = torch.argmax(
    end_logits,
    dim=-1,
).item()


print("\nSTART POSITION:")
print(start_position)

print("\nEND POSITION:")
print(end_position)


# ---------------------------------------------------------
# 8. GET TOKEN IDS
# ---------------------------------------------------------

input_ids = inputs["input_ids"][0]


# ---------------------------------------------------------
# 9. EXTRACT ANSWER TOKENS
# ---------------------------------------------------------

answer_tokens = input_ids[
    start_position:end_position + 1
]


# ---------------------------------------------------------
# 10. CONVERT TOKENS BACK TO TEXT
# ---------------------------------------------------------

answer = tokenizer.decode(
    answer_tokens,
    skip_special_tokens=True,
)


print("\nANSWER:")
print(answer)