import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model_fp32 = AutoModelForCausalLM.from_pretrained(
    "distilgpt2"
).to("mps")

model_fp16 = AutoModelForCausalLM.from_pretrained(
    "distilgpt2"
).to("mps").half()

text = "The future of AI is"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

inputs = {
    k: v.to("mps")
    for k, v in inputs.items()
}

def benchmark(model, inputs, runs=20, warmup=5):
    model.eval()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(**inputs)

    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    start = time.perf_counter()

    for _ in range(runs):
        with torch.no_grad():
            _ = model(**inputs)

    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    end = time.perf_counter()

    return (end - start) / runs

fp32_time = benchmark(model_fp32, inputs)
fp16_time = benchmark(model_fp16, inputs)

print(f"FP32: {fp32_time * 1000:.2f} ms")
print(f"FP16: {fp16_time * 1000:.2f} ms")

print(f"Speedup: {fp32_time / fp16_time:.2f}x")

def generate_benchmark(model, inputs, runs=5):
    model.eval()

    # Warmup
    for _ in range(2):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

    torch.mps.synchronize()

    start = time.perf_counter()

    for _ in range(runs):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

    torch.mps.synchronize()

    end = time.perf_counter()

    return (end - start) / runs

fp32_gen = generate_benchmark(model_fp32, inputs)
fp16_gen = generate_benchmark(model_fp16, inputs)

print(f"FP32 generation: {fp32_gen:.3f} sec")
print(f"FP16 generation: {fp16_gen:.3f} sec")
print(f"Speedup: {fp32_gen / fp16_gen:.2f}x")