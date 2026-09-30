import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


# ---------------------------------------------------------
# 1. LOAD TOKENIZER
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------
# 2. LOAD MODEL
# ---------------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)


# ---------------------------------------------------------
# 3. INPUT TEXT
# ---------------------------------------------------------

text = ["I love this product.", "I hate this product."]


# ---------------------------------------------------------
# 4. TOKENIZE
# ---------------------------------------------------------

inputs = tokenizer(
    text,
    return_tensors="pt",
)

print("TOKENIZER OUTPUT:")
print(inputs)


# ---------------------------------------------------------
# 5. RUN MODEL
# ---------------------------------------------------------

with torch.no_grad():
    outputs = model(**inputs)


# ---------------------------------------------------------
# 6. RAW MODEL OUTPUT
# ---------------------------------------------------------

print("\nMODEL OUTPUT:")
print(outputs)


# ---------------------------------------------------------
# 7. LOGITS
# ---------------------------------------------------------

logits = outputs.logits

print("\nLOGITS:")
print(logits)


# ---------------------------------------------------------
# 8. CONVERT LOGITS TO PROBABILITIES
# ---------------------------------------------------------

probabilities = torch.softmax(logits, dim=-1)

print("\nPROBABILITIES:")
print(probabilities)


# ---------------------------------------------------------
# 9. GET PREDICTED CLASS
# ---------------------------------------------------------

predicted_class_ids = torch.argmax(probabilities, dim=-1).tolist()


print("\nPREDICTED CLASS IDs:")
print(predicted_class_ids)


# ---------------------------------------------------------
# 10. CONVERT CLASS IDs TO LABELS
# ---------------------------------------------------------

# Map each class ID to its text label using a list comprehension
labels = [model.config.id2label[class_id] for class_id in predicted_class_ids]

print("\nPREDICTED LABELS:")
for i, label in enumerate(labels):
    print(f"Text {i+1}: {label}")