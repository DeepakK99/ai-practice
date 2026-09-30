from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np


# ---------------------------------------------------------
# 1. LABEL DEFINITIONS
# ---------------------------------------------------------

label_names = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
]

label2id = {
    label: i
    for i, label in enumerate(label_names)
}

id2label = {
    i: label
    for label, i in label2id.items()
}


# ---------------------------------------------------------
# 2. SMALL NER DATASET
# ---------------------------------------------------------

data = {
    "tokens": [
        ["John", "works", "at", "Amazon", "in", "Seattle"],
        ["Mary", "works", "at", "Google", "in", "London"],
        ["Rahul", "lives", "in", "Bangalore"],
        ["Alice", "works", "for", "Microsoft", "in", "Delhi"],
    ],
    "ner_tags": [
        [1, 0, 0, 3, 0, 5],
        [1, 0, 0, 3, 0, 5],
        [1, 0, 0, 5],
        [1, 0, 0, 3, 0, 5],
    ],
}

dataset = Dataset.from_dict(data)


# ---------------------------------------------------------
# 3. LOAD TOKENIZER
# ---------------------------------------------------------

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# ---------------------------------------------------------
# 4. TOKENIZE + ALIGN LABELS
# ---------------------------------------------------------

def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
    )
    
    all_labels = []

    for batch_index, labels in enumerate(
        examples["ner_tags"]
    ):

        word_ids = tokenized_inputs.word_ids(
            batch_index
        )

        previous_word_id = None
        label_ids = []

        for word_id in word_ids:

            # Special token
            if word_id is None:
                label_ids.append(-100)

            # First token of a word
            elif word_id != previous_word_id:
                label_ids.append(
                    labels[word_id]
                )

            # Additional subword token
            else:
                label_ids.append(-100)

            previous_word_id = word_id

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    
    return tokenized_inputs


tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
)


# ---------------------------------------------------------
# 5. LOAD DISTILBERT + TOKEN CLASSIFICATION HEAD
# ---------------------------------------------------------

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
)


# ---------------------------------------------------------
# 6. TRAINING CONFIGURATION
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir="./distilbert-ner",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    logging_steps=1,
    report_to="none",
)


# ---------------------------------------------------------
# 7. TRAINER
# ---------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)


# ---------------------------------------------------------
# 8. TRAIN
# ---------------------------------------------------------

trainer.train()


# ---------------------------------------------------------
# 9. SAVE
# ---------------------------------------------------------

trainer.save_model(
    "./distilbert-ner-final"
)

tokenizer.save_pretrained(
    "./distilbert-ner-final"
)


print("\nModel saved to ./distilbert-ner-final")