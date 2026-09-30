from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The future of AI is"

inputs = tokenizer(prompt, return_tensors="pt")

####################
# generation
####################
# deterministic
# output = model.generate(
#     **inputs,
#     max_new_tokens=30,
#     do_sample=False
# )

# sampling
# output = model.generate(
#     **inputs,
#     max_new_tokens=30,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.9
# )

# text = tokenizer.decode(output[0])

# print(repr(text))

# print(next(model.parameters()).device)

####################
# size calculation
####################
# import torch

# print("CUDA available:", torch.cuda.is_available())
# print("Model dtype:", next(model.parameters()).dtype)
# print("Model device:", next(model.parameters()).device)

# num_params = sum(p.numel() for p in model.parameters())

# print("Parameters:", num_params)
# print("Parameters (millions):", num_params / 1e6)

# memory_gb = num_params * 4 / (1024 ** 3)

# print("Approx FP32 weight memory:", memory_gb, "GB")

####################
# mps check (gpu)
####################
# import torch

# print("MPS available:", torch.backends.mps.is_available())
# print("MPS built:", torch.backends.mps.is_built())

print("Before:")
print("dtype:", next(model.parameters()).dtype)
print("device:", next(model.parameters()).device)

model_fp16 = model.half()

model_fp16 = model_fp16.to("mps")

print("\nAfter:")
print("dtype:", next(model_fp16.parameters()).dtype)
print("device:", next(model_fp16.parameters()).device)