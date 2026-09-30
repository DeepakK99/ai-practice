import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

def setup():

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(device)

    model.eval()

    print("Model loaded.")

    return tokenizer, model, device, MODEL_NAME
