import torch

def simulate_8bit_quantization(tensor):
    # 1. Find the maximum absolute value in the tensor to set the bounds
    max_val = torch.max(torch.abs(tensor))
    
    # 2. Calculate the "Scale factor" 
    # We want to map the max value perfectly to the edge of the int8 boundary (127)
    scale = max_val / 127.0
    
    # 3. QUANTIZATION: Scale the numbers up, and round them to whole integers
    # We clip them to ensure no value goes outside the [-128, 127] boundaries
    quantized = torch.clamp(torch.round(tensor / scale), -128, 127)
    
    # 4. DE-QUANTIZATION: Convert them back to floats so we can see the damage
    dequantized = quantized * scale
    
    return quantized, dequantized, scale

# ---------------------------------------------------------
# Test it with a mock weights tensor (e.g., neural network layer weights)
# ---------------------------------------------------------
# Let's create a small tensor of 5 random weights
original_weights = torch.tensor([0.1523, -0.8941, 2.4512, -1.1204, 0.0345])

quant_matrix, simulated_weights, scale_factor = simulate_8bit_quantization(original_weights)

print(f"Original Weights (float32):\n {original_weights}\n")

print(f"1. Quantized Matrix (Stored as whole Int8 numbers):\n {quant_matrix}")
print(f"   (Calculated Scale Factor: {scale_factor:.6f})\n")

print(f"2. De-quantized Weights (What the model actually runs on):\n {simulated_weights}\n")

# ---------------------------------------------------------
# Measure the Quantization Error (Accuracy Loss)
# ---------------------------------------------------------
error = original_weights - simulated_weights
print("--- ACCURACY LOSS ANALYSIS ---")
print(f"Difference per element:\n {error}")
print(f"Average absolute rounding error: {torch.mean(torch.abs(error)):.6f}")
