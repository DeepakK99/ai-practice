import torch

weights = torch.tensor([
    -8.0, -6.0, -1.0, 0.5,
     0.1,  0.5,  0.7, 0.8
])

block_size = 4
qmax = 7

blocks = weights.view(-1, block_size)

scales = blocks.abs().max(dim=1).values / qmax

quantized = torch.round(blocks / scales[:, None])

reconstructed = quantized * scales[:, None]

print("Original:")
print(blocks)

print("\nScales:")
print(scales)

print("\nQuantized:")
print(quantized)

print("\nReconstructed:")
print(reconstructed)

print("\nError:")
print(blocks - reconstructed)