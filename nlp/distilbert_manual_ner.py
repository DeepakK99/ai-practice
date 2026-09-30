import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


MODEL_NAME = "elastic/distilbert-base-uncased-finetuned-conll03-english"


# ---------------------------------------------------------
# 1. LOAD TOKENIZER
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------
# 2. LOAD NER MODEL
# ---------------------------------------------------------

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME
)


# ---------------------------------------------------------
# 3. INPUT
# ---------------------------------------------------------

text = "John alter works at Amazon in Bangalore."


# ---------------------------------------------------------
# 4. TOKENIZE
# ---------------------------------------------------------

inputs = tokenizer(
    text,
    return_tensors="pt",
)

print("TOKENS:")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))


# ---------------------------------------------------------
# 5. RUN MODEL
# ---------------------------------------------------------

with torch.no_grad():
    outputs = model(**inputs)


# ---------------------------------------------------------
# 6. RAW LOGITS
# ---------------------------------------------------------

logits = outputs.logits

print("\nLOGITS SHAPE:")
print(logits.shape)


# ---------------------------------------------------------
# 7. GET PREDICTED LABEL FOR EACH TOKEN
# ---------------------------------------------------------

predicted_ids = torch.argmax(
    logits,
    dim=-1,
)


# ---------------------------------------------------------
# 8. MAP IDS TO LABELS
# ---------------------------------------------------------

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("\nTOKEN PREDICTIONS:")

for token, label_id in zip(
    tokens,
    predicted_ids[0],
):
    label = model.config.id2label[label_id.item()]

    print(
        f"{token:15} -> {label}"
    )

print(tokenizer.decode([inputs["input_ids"][0][1:3]]))

print(
    tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )
)

print(model.config.id2label)