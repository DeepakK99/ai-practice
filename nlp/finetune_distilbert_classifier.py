from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------
# 1. SMALL TRAINING DATASET
# ---------------------------------------------------------

data = {
    "text": [
        "I was charged twice for my order",
        "Why was I charged an extra fee?",
        "My payment was declined",
        "I cannot log into my account",
        "I forgot my password",
        "I need to change my email address",
        "The application keeps crashing",
        "The website is showing an error",
        "The app is extremely slow",
        "I was charged twice",
        "My password reset is not working",
        "The mobile application crashes when I open it",
    ],
    "label": [
        0, 0, 0,
        1, 1, 1,
        2, 2, 2,
        0, 1, 2,
    ],
}

dataset = Dataset.from_dict(data)


# ---------------------------------------------------------
# 2. LABEL DEFINITIONS
# ---------------------------------------------------------

id2label = {
    0: "BILLING",
    1: "ACCOUNT",
    2: "TECHNICAL",
}

label2id = {
    "BILLING": 0,
    "ACCOUNT": 1,
    "TECHNICAL": 2,
}


# ---------------------------------------------------------
# 3. LOAD TOKENIZER
# ---------------------------------------------------------

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)


# ---------------------------------------------------------
# 4. TOKENIZE DATASET
# ---------------------------------------------------------

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
)


# ---------------------------------------------------------
# 5. LOAD DISTILBERT + CLASSIFICATION HEAD
# ---------------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)


# ---------------------------------------------------------
# 6. METRICS
# ---------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "f1": f1_score(
            labels,
            predictions,
            average="weighted",
        ),
    }


# ---------------------------------------------------------
# 7. TRAINING CONFIGURATION
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir="./distilbert-support",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=1,
    report_to="none",
)


# ---------------------------------------------------------
# 8. TRAINER
# ---------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    compute_metrics=compute_metrics,
)


# ---------------------------------------------------------
# 9. TRAIN
# ---------------------------------------------------------

trainer.train()


# ---------------------------------------------------------
# 10. SAVE MODEL
# ---------------------------------------------------------

trainer.save_model(
    "./distilbert-support-final"
)

tokenizer.save_pretrained(
    "./distilbert-support-final"
)


# ---------------------------------------------------------
# 11. TEST
# ---------------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    "./distilbert-support-final"
)

tokenizer = AutoTokenizer.from_pretrained(
    "./distilbert-support-final"
)

test_texts = [
    "My card was charged twice",
    "I forgot my password",
    "The website keeps crashing",
]

inputs = tokenizer(
    test_texts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)

import torch

model.eval() 
predictions = None
with torch.inference_mode():
    outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

print("\nPREDICTIONS:")

for text, prediction in zip(
    test_texts,
    predictions,
):
    print(
        f"{text} -> {id2label[int(prediction)]}"
    )