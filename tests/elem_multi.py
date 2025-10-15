import torch
from einops import einsum, rearrange

d_model = 4
d_ff = 6

# Create a weight matrix
W1 = torch.randn(d_ff, d_model)  # Shape: (6, 4)
x = torch.randn(d_model)          # Shape: (4,)

# Method 1: Mathematical notation W1 @ x
result1 = W1 @ x  # Shape: (6,)

# Method 2: PyTorch Linear convention x @ W1.T
result2 = x @ W1.T  # Shape: (6,)

# Method 3: Your einsum (x first, then W1)
# result3 = torch.einsum("d_model, d_ff d_model -> d_ff", x, W1)
result3 = einsum(x, W1, "d_model, d_ff d_model -> d_ff")

print("All equal?", torch.allclose(result1, result2) and torch.allclose(result2, result3))
# Output: All equal? True