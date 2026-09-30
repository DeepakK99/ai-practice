import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Setup tokenizer with required parameters for causal batching
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"      # CRITICAL for batch inference
tokenizer.pad_token = tokenizer.eos_token  # CRITICAL: Define padding token

# 2. Load the SINGLE instance of the model into memory
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)

def generate_batch(prompt, concurrency, max_new_tokens=50):
    # Create an explicit batch by repeating the prompt for the concurrency level
    batch_prompts = [prompt] * concurrency

    # Tokenizer automatically creates a batched tensor: [batch_size, sequence_length]
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True # Tells tokenizer to equalize prompt lengths using pad_token
    ).to(device)

    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        # The single model processes the entire batch in a single forward pass
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id # Explicitly pass pad token ID
        )

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    # Calculate total generated tokens across the entire batch
    total_input_tokens = inputs["input_ids"].shape[-1] * concurrency
    total_output_tokens = outputs.shape[-1] * concurrency
    generated_tokens = total_output_tokens - total_input_tokens

    return elapsed, generated_tokens


prompt = "Explain retrieval augmented generation."

# Test the throughput scaling as concurrency increases
for concurrency in [1, 2, 4, 8]:

    # Warm-up run to ensure the Metal graph compilation doesn't skew numbers
    if concurrency == 1:
        _ = generate_batch(prompt, concurrency=1, max_new_tokens=5)

    elapsed_time, total_tokens = generate_batch(
        prompt, 
        concurrency=concurrency, 
        max_new_tokens=50
    )

    print(f"\nConcurrency (Batch Size): {concurrency}")
    print(f"Total wall time: {elapsed_time:.3f}s")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Aggregate throughput: {total_tokens / elapsed_time:.2f} tok/s")
