import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
).to(device)


model.eval()

prompt = "Explain retrieval augmented generation."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)

for max_new_tokens in [10, 50, 100, 200]:

    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    generated_tokens = (
        output.shape[-1] - inputs["input_ids"].shape[-1]
    )

    print(
        f"max_new_tokens={max_new_tokens:3d} | "
        f"generated={generated_tokens:3d} | "
        f"time={elapsed:.3f}s | "
        f"tokens/sec={generated_tokens / elapsed:.2f}"
    )
print("\n===========================\n")

########
# TTFT #
########
prompt = "Explain retrieval augmented generation."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)

if device == "mps":
    torch.mps.synchronize()

start = time.perf_counter()

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
    )

if device.type == "mps":
    torch.mps.synchronize()

ttft = time.perf_counter() - start

print(f"Approx TTFT: {ttft:.3f}s")