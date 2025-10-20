import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Args:
            theta: Base frequency (Θ) used in RoPE
            d_k: Dimension of key/query vectors
            max_seq_len: Maximum sequence length
            device: torch.device to store precomputed cos/sin tables
        """
        # import pdb;pdb.set_trace()
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Compute frequency base for each pair of dimensions
        # RoPE is applied pairwise, so we take half the dimension
        half_dim = d_k // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, device=device).float() / half_dim))

        # Create the position index [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, device=device).float()

        # Compute sinusoidal values (shape: [max_seq_len, half_dim])
        freqs = torch.einsum("i,j->ij", positions, inv_freq)  # outer product
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # Register as buffers (non-trainable tensors)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor x.

        Args:
            x: Tensor of shape (..., seq_len, d_k)
            token_positions: Tensor of shape (..., seq_len)
        Returns:
            Tensor of same shape (..., seq_len, d_k)
        """
        # x shape: (..., seq_len, d_k)
        *batch_dims, seq_len, d_k = x.shape
        assert d_k == self.d_k, "Input last dimension must match d_k"

        # Get cos/sin for the given token positions
        # token_positions shape: (..., seq_len)
        cos = self.cos_cached[token_positions]  # (..., seq_len, half_dim)
        sin = self.sin_cached[token_positions]

        # Expand for broadcasting to match x
        while len(cos.shape) < len(x.shape):
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # Split x into even/odd parts
        x1, x2 = x[..., ::2], x[..., 1::2]
        # cal dims myself to verify
        # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        
        x_rotated = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            dim=-1
        )

        # Merge last two dims back to (..., seq_len, d_k)
        return x_rotated.flatten(-2)