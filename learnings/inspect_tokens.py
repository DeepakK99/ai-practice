from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

texts = [
    "I love machine learning",
    "I love machinelearning",
    "I am learning machine learning",
    "The quick brown fox jumps over the lazy dog.",
    "Tokenization is surprisingly important!",
    "unhappiness",
    "playing",
    "play",
]

for text in texts:
    print("\nText: ", text)
    tokens = tokenizer.tokenize(text)
    print("No. of Tokens: ", len(tokens))
    ids = tokenizer.encode(text)
    print([f"{id}: {token}" for id, token in zip(ids, tokens)])
