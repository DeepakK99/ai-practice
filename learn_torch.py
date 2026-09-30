import torch

# 1. Create a tensor from a Python list (like Token IDs)
token_ids = torch.tensor([[101, 2054, 2003, 102]])

# 2. Crucial Inspection Tools (You will use these daily)
print("Shape:", token_ids.shape)  # Dimensions of the grid (e.g., torch.Size([1, 4]))
print("Data Type:", token_ids.dtype)  # Data type (e.g., torch.int64 for IDs)
print("Device:", token_ids.device)  # Where the data lives (CPU or GPU)

# Imagine this is your model's output logits: [Batch, Seq_Len, Vocab]
logits = torch.randn(1, 3, 50257) 
print(logits)

# Get the logits for ALL batches, ALL sequences, but only the 5th word in vocab
word_five = logits[:, :, 5] 
print(word_five)

# Get the logits for the VERY LAST token in the sequence (Standard for LLMs)
# ":" means "all batches", "-1" means "last item in sequence"
next_token_logits = logits[0, -1, :] 
print("Sliced Shape:", next_token_logits.shape) # Output: torch.Size([50257])
