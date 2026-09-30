# ==============================================================================
# LORA TRAINING & LOSS MASKING REFERENCE
# ==============================================================================
# 1. FORWARD PASS: The model processes the entire prompt (User + Assistant) 
#    to generate contextual vectors and logits. User context is required so 
#    the model understands the question.
#
# 2. GRADIENT MASKING: The data collator sets all User token targets to `-100`. 
#    PyTorch's loss function (`ignore_index=-100`) completely throws away 
#    the predictions for these tokens.
#
# 3. BACKPROPAGATION: Gradients are calculated and saved ONLY for the Assistant's 
#    response. The optimizer updates our 540k LoRA parameters to learn how to 
#    ANSWER correctly, while completely ignoring the user's phrasing.
# ==============================================================================

import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from trl import SFTTrainer


# ============================================================
# 1. Model
# ============================================================

model_name = "Qwen/Qwen2.5-0.5B-Instruct"


# ============================================================
# 2. Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 3. 4-bit quantization
# ============================================================

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,

    # 4-bit storage format
    bnb_4bit_quant_type="nf4",

    # Computation happens in BF16
    bnb_4bit_compute_dtype=torch.bfloat16,

    # Additional memory optimization
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 4. Load quantized base model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
)

# Important for training
model.config.use_cache = False


# ============================================================
# 5. LoRA configuration
# ============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)


# ============================================================
# 6. Attach LoRA adapters
# ============================================================

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()


# ============================================================
# 7. Training dataset
# ============================================================

data = [
    {
        "messages": [
            {
                "role": "user",
                "content": "What is RAG?",
            },
            {
                "role": "assistant",
                "content": (
                    "RAG stands for Retrieval-Augmented Generation. "
                    "It retrieves relevant information and provides "
                    "that information to an LLM as context."
                ),
            },
        ]
    },

    {
        "messages": [
            {
                "role": "user",
                "content": "What is an embedding?",
            },
            {
                "role": "assistant",
                "content": (
                    "An embedding is a numerical vector representation "
                    "of data that captures semantic relationships."
                ),
            },
        ]
    },
]

dataset = Dataset.from_list(data)


# ============================================================
# 8. Training arguments
# ============================================================

training_args = TrainingArguments(
    output_dir="./qwen-qlora",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    logging_steps=10,

    save_steps=100,

    bf16=True,

    optim="paged_adamw_8bit",

    report_to="none",
)


# ============================================================
# 9. SFT Trainer
# ============================================================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,

    processing_class=tokenizer,

    peft_config=lora_config,

    args=training_args,
)


# ============================================================
# 10. Train
# ============================================================

trainer.train()


# ============================================================
# 11. Save LoRA adapter
# ============================================================

model.save_pretrained(
    "./qwen-qlora-adapter"
)

tokenizer.save_pretrained(
    "./qwen-qlora-adapter"
)



# ============================================================
#  12. using the saved adapter
# ============================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Define the paths (Use the exact name or path of your base model)
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Replace with your specific Qwen base model
ADAPTER_PATH = "./qwen-qlora-adapter"

# 2. Load the base tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,  # Match the bf16=True from your training args
    device_map="auto"            # Automatically puts it on your GPU
)

# 3. Load and merge the LoRA adapter onto the base model
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# 4. Put the model in evaluation mode (turns off dropout!)
model.eval()