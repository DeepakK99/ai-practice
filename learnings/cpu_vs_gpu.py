import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Targeting device: {device.upper()}")

# 1. Tokenize the input prompt
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("The future of AI is highly dependent on", return_tensors="pt")

# 2. Load the model directly as float16 (half precision) and send it straight to GPU
# torch_dtype=torch.float16 ensures we never load the heavy float32 weights into RAM
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16
).to(device)

# 3. Prepare inputs for the GPU
inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

# 4. GPU Warm-up Pass (Crucial for accurate GPU benchmarking)
with torch.no_grad():
    _ = model(**inputs_gpu)

# 5. Measure the optimized generation speed
start_time = time.time()
with torch.no_grad():
    outputs = model(**inputs_gpu)
gpu_duration = time.time() - start_time

print(f"\n⚡ Half-Precision GPU (MPS) Execution Time: {gpu_duration:.4f} seconds")

# 6. Let's see the logits shape to confirm everything worked!
print(f"Logits Shape: {outputs.logits.shape}")
print(f"Logits Data Type: {outputs.logits.dtype}") # Should show float16!


# manual clearing
# del model
# if device == "mps":
#     torch.mps.empty_cache()
