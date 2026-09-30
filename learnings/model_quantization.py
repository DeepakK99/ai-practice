import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

device = torch.device("mps")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)

###################################
# weights memory calculation
###################################
num_params = sum(p.numel() for p in model.parameters())

print("Parameters:", num_params)
print("Parameters (M):", num_params / 1e6)

print("Dtype:", next(model.parameters()).dtype)
print("Device:", next(model.parameters()).device)

memory_gb = num_params * 2 / (1024**3)

print("Approx FP16 weight memory:", memory_gb, "GB")

_4_bit_memory_gb = num_params * 0.5 / (1024 ** 3)
print("Approx 4-bit weight memory:", _4_bit_memory_gb, "GB")


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Explain Retrieval-Augmented Generation(RAG) in one paragraph."
    },
]

# inputs = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(device)


# print("Input shape:", inputs["input_ids"].shape)

###################################
# output generation
###################################
# with torch.no_grad():
#     output_ids = model.generate(
#         **inputs,
#         max_new_tokens=100,
#         do_sample=False,
#     )
# print("Output shape:", output_ids.shape)

# generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

# response = tokenizer.decode(
#     generated_ids,
#     skip_special_tokens=True
# )

# print("Respone:", response)

###################################
# peeking in the model configs
###################################
# print("Model:", model)
# print("Config:", model.config)
# print("Layer_0:", model.model.layers[0])
# print("Layer_0_self_attn:", model.model.layers[0].self_attn)
# print("MLP:", model.model.layers[0].mlp)

# attn = model.model.layers[0].self_attn

# print("Q:", attn.q_proj.weight.shape)
# print("K:", attn.k_proj.weight.shape)
# print("V:", attn.v_proj.weight.shape)
# print("O:", attn.o_proj.weight.shape)


# Calculate head dimension: hidden_size (896) / num_attention_heads (14)
# head_dim = model.config.hidden_size // model.config.num_attention_heads

# print("--- ATTENTION HEAD ARCHITECTURE ---")
# print("Hidden Size (Total Channels): ", model.config.hidden_size)
# print("Number of Query Heads(Attention heads):       ", model.config.num_attention_heads)
# print("Head Dimension (Per Head): ", head_dim)

# print("\n--- GROUPED-QUERY ATTENTION (GQA) ---")
# print("Number of Key/Value(KV) Heads: ", model.config.num_key_value_heads)

# # Calculate how many query heads share a single KV head (14 / 2 = 7)
# heads_per_group = model.config.num_attention_heads // model.config.num_key_value_heads
# print(f"GQA Ratio: 7 Query heads share {heads_per_group} KV head(s)")

###################################
# mapping input through first layer
###################################
# layer = model.model.layers[0]

# print("Input layer norm:", layer.input_layernorm)
# print("Attention:", layer.self_attn)
# print("MLP:", layer.mlp)

# print("Hidden size:", model.config.hidden_size)
# print("Q heads:", model.config.num_attention_heads)
# print("KV heads:", model.config.num_key_value_heads)
# print("Layers:", model.config.num_hidden_layers)

# hidden_states = model.model.embed_tokens(inputs["input_ids"])

# print("Embedding output:", hidden_states.shape)

# attn = model.model.layers[0].self_attn

# Q = attn.q_proj(hidden_states)
# K = attn.k_proj(hidden_states)
# V = attn.v_proj(hidden_states)

# print("Q:", Q.shape)
# print("K:", K.shape)
# print("V:", V.shape)