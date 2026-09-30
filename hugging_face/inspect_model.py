def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable

def inspect_model(model, tokenizer):

    config = model.config

    total_params, trainable_params = count_parameters(model)

    print("========== MODEL ==========")
    print("Architecture:", model.__class__.__name__)
    print("Model type:", config.model_type)

    print("\n========== PARAMETERS ==========")
    print("Total:", total_params)
    print("Total (M):", total_params / 1e6)

    print("Trainable:", trainable_params)
    print("Trainable (M):", trainable_params / 1e6)

    print("\n========== ARCHITECTURE ==========")
    print("Layers:", config.num_hidden_layers)
    print("Hidden size:", config.hidden_size)

    print("Attention heads:", config.num_attention_heads)

    if hasattr(config, "num_key_value_heads"):
        print("KV heads:", config.num_key_value_heads)

    if hasattr(config, "head_dim"):
        print("Head dimension:", config.head_dim)

    print("\n========== VOCAB / CONTEXT ==========")
    print("Vocabulary size:", config.vocab_size)

    if hasattr(config, "max_position_embeddings"):
        print(
            "Max position embeddings:",
            config.max_position_embeddings
        )

    print("\n========== RUNTIME ==========")

    dtypes = set(
        p.dtype
        for p in model.parameters()
    )

    devices = set(
        p.device
        for p in model.parameters()
    )

    print("Dtype:", dtypes)
    print("Device:", devices)

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inspect_model(model, tokenizer)